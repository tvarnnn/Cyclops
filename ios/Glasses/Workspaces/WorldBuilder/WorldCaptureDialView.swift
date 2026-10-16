//
//  WorldCaptureDialView.swift
//  Glasses
//
//  The FOW capture dial, drawn (FOW-RING-SPEC §4): twelve rounded annular
//  segments with constant 3 pt gaps on an always-dark stage, start fixed at
//  12 o'clock (never turned), the piece's photos in its filled segments, the
//  count in the centre, the amber "scan here" outline on the largest gap,
//  and mini dials for the other pieces. The header above the stage (the
//  title, and the Dial|Map control with IOS_FOW_MAP_TAB=on), the one-line
//  footer and the "Recently captured" strip under it follow the system
//  theme. Nothing here announces, plays a sound or buzzes.
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
    /// walking card's 361 × 271 stage. `verticalRoom` is the 40: less when a
    /// row of mini dials under the ring takes the stage's bottom.
    init(stage size: CGSize, verticalRoom: CGFloat = 40) {
        diameter = max(96, min(size.height - verticalRoom, size.width - 120, 300))
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

// MARK: - The stage: the dial or the map; the header that switches them

/// The capture card's header (§5): one fixed title, and the
/// Dial|Map control on its trailing side once there is a map to switch to --
/// above the stage, so it never sits on the ring.
struct WorldCaptureDialHeader: View {
    @Binding var tab: WorldPanelTab
    let showsToggle: Bool
    /// After Stop the control stays in place, on Dial, and does nothing:
    /// the panel draws no map once the walk has stopped.
    var togglable = true

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            Text(WorldCaptureDialCopy.title)
                .font(.headline)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
                .accessibilityAddTraits(.isHeader)
                .accessibilityIdentifier("wb-panel-dial-title")
            Spacer(minLength: 0)
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
                .disabled(!togglable)
                .accessibilityIdentifier("wb-panel-tab")
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// The walking stage (§4 Layout, §5): the dial, or on the Map tab the
/// current coverage map, with a photo plan when available.
struct WorldCaptureDialStage<MapContent: View>: View {
    let dial: WorldCaptureDial
    /// The CURRENT receipt (never a held one), or `nil` before the first landing.
    let receipt: WorldCoverageReceipt?
    /// The photos the dial shows: this landing's, and those kept for its hero.
    let photos: WorldCaptureDialPhotos
    let tab: WorldPanelTab
    @ViewBuilder let map: () -> MapContent

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        ZStack {
            if receipt != nil, tab == .map {
                map()
                    .transition(.opacity)
            } else {
                WorldCaptureDialView(dial: dial, receipt: receipt, photos: photos)
                    .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: reduceMotion ? 0.15 : 0.25), value: tab)
    }
}

// MARK: - The dial

struct WorldCaptureDialView: View {
    let dial: WorldCaptureDial
    let receipt: WorldCoverageReceipt?
    let photos: WorldCaptureDialPhotos
    /// The look-here segment, the walk cue and the mini dials: off for the
    /// dimmed final dial after Stop, where there is nowhere left to look.
    var guides = true

    /// The mini-dial row's height, under the ring.
    static let miniRow: CGFloat = 28

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.colorSchemeContrast) private var contrast
    @Environment(\.accessibilityVoiceOverEnabled) private var voiceOver
    @Environment(\.dynamicTypeSize) private var typeSize

    @State private var selected: WorldDialThumbnail?

    private var increased: Bool { contrast == .increased }

    var body: some View {
        GeometryReader { proxy in
            // Other pieces get their own row under the ring, never on it:
            // the ring is laid out in the stage above that row.
            let others = guides ? dial.miniPieces : []
            let row: CGFloat = others.isEmpty ? 0 : Self.miniRow + 12
            let region = CGSize(width: proxy.size.width, height: proxy.size.height - row)
            let geometry = WorldDialGeometry(stage: region, verticalRoom: others.isEmpty ? 40 : 28)
            ZStack {
                WorldDialStyle.stage
                dialFace(geometry)
                    .frame(width: geometry.diameter, height: geometry.diameter)
                    .position(x: proxy.size.width / 2, y: region.height / 2)
                if !others.isEmpty {
                    minis(others)
                        .frame(height: Self.miniRow)
                        .padding(.horizontal, 12)
                        .padding(.bottom, 8)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
                }
            }
        }
        .environment(\.colorScheme, .dark)
        // An open sheet never outlives its photo.
        .onChange(of: photos) { _, photos in
            if let shown = selected, !photos.bySector.values.contains(where: { $0.digest == shown.digest }) {
                selected = nil
            }
        }
        .sheet(item: $selected) { photo in
            WorldCaptureDialPhotoSheet(photo: photo)
        }
    }

    // MARK: The face

    private func dialFace(_ geometry: WorldDialGeometry) -> some View {
        ZStack {
            ring(geometry)
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
        .background {
            // The dial's one element (§4): a leaf under the drawing, so a
            // photo's own tap is never taken by it.
            Circle()
                .fill(Color.white.opacity(0.001))
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(WorldCaptureDialCopy.accessibilityLabel(dial))
                .accessibilityValue(WorldCaptureDialCopy.accessibilityValue(dial))
                .accessibilityIdentifier("wb-panel-dial")
                .accessibilityActions {
                    // After Stop the dimmed dial is never touched: no actions.
                    ForEach(guides ? photoPositions : [], id: \.self) { position in
                        Button(WorldCaptureDialCopy.photo(at: position)) { open(sector: dial.sector(at: position)) }
                    }
                }
                .accessibilitySortPriority(2)
        }
    }

    /// Display positions holding a photo, clockwise from 12 o'clock.
    private var photoPositions: [Int] {
        (0..<WorldCaptureDial.sectors).filter { photos.bySector[dial.sector(at: $0)] != nil }
    }

    private func ring(_ geometry: WorldDialGeometry) -> some View {
        let lookHere = guides ? dial.lookHere : nil
        let marks = geometry.diameter + WorldDialGeometry.markRoom * 2
        return ZStack {
            ForEach(0..<WorldCaptureDial.sectors, id: \.self) { sector in
                segment(sector, geometry: geometry)
                    .frame(width: geometry.diameter, height: geometry.diameter)
            }
        }
        .frame(width: geometry.diameter, height: geometry.diameter)
        .overlay {
            if let lookHere {
                WorldDialLookHere(first: dial.sector(at: lookHere.positions[0]), count: lookHere.positions.count,
                                  radius: geometry.radius)
                    .frame(width: marks, height: marks)
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)
            }
        }
    }

    private func segment(_ sector: Int, geometry: WorldDialGeometry) -> some View {
        let state = dial.evidence(sector: sector)
        let position = dial.position(of: sector)
        let shape = AnnularSector(first: sector)
        let photo = photos.bySector[sector]
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

    /// The photo fills its wedge (`photoSide` ≥ 1.15 of the segment depth),
    /// upright (the ring never turns), with no darkening over it and a slight
    /// lift: a one-view segment's photo stays faded and grey, so the two
    /// states still read. A tap opens it large.
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
            .saturation(weak ? 0.2 : 1.05)
            .brightness(weak ? 0 : 0.06)
            .opacity(weak ? 0.5 : 1)
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
            // one element (§4). Other tools (Voice Control, tests) see them,
            // live only: after Stop the dimmed dial is never read or touched.
            .accessibilityHidden(voiceOver || !guides)
    }

    /// A captured segment with no photo is lit teal, a little brighter
    /// towards the rim; under a photo the stage.
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

    /// One row under the ring: two pieces, then "+N more".
    private func minis(_ others: [WorldCaptureDial.Piece]) -> some View {
        HStack(alignment: .center, spacing: 14) {
            ForEach(others.prefix(2), id: \.referenceSegment) { piece in
                WorldDialMini(piece: piece)
            }
            if others.count > 2 {
                Text(WorldCaptureDialCopy.more(others.count - 2))
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(Color.white.opacity(0.8))
                    .lineLimit(1)
                    .fixedSize()
                    .dynamicTypeSize(...DynamicTypeSize.xLarge)
            }
        }
        .allowsHitTesting(false)
    }

    private func open(sector: Int) {
        selected = photos.bySector[sector]
    }
}

// MARK: - The marks

/// 12 o'clock: a 2 pt white rim over the top segment with a notch pointing
/// in (§4), and the tiny "start" caption just past the rim's end: the top is
/// the way the wearer faced when the piece began. It never turns.
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
            // Clear of the rim's end and of the ring below it.
            let caption = Text(WorldCaptureDialCopy.start)
                .font(.system(size: 11, weight: .semibold, design: .rounded))
                .foregroundColor(.white.opacity(0.8))
            context.draw(caption, at: CGPoint(x: centre.x + (radius + 4) * sin(11 * .pi / 180) + 6,
                                              y: centre.y - radius - 7),
                         anchor: .leading)
        }
        .allowsHitTesting(false)
    }
}

/// The look-here emphasis (§4, polished): ONE next-best segment, or two
/// for a long gap -- a 2.5 pt amber outline with an 8 pt glow, pulsing
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
}

/// Another piece, colour only and not interactive (§3.7), in the row under
/// the ring: a 28 pt ring and "Piece 2 · 3/12" beside it.
struct WorldDialMini: View {
    let piece: WorldCaptureDial.Piece

    var body: some View {
        HStack(alignment: .center, spacing: 6) {
            ZStack {
                ForEach(0..<WorldCaptureDial.sectors, id: \.self) { sector in
                    AnnularSector(first: sector, innerRatio: 9.0 / 14, gap: 1.2, cornerRadius: 1)
                        .fill(fill(piece.sectors[sector]))
                }
            }
            .frame(width: WorldCaptureDialView.miniRow, height: WorldCaptureDialView.miniRow)
            Text(WorldCaptureDialCopy.miniLabel(piece))
                .font(.caption2.weight(.semibold).monospacedDigit())
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

/// Under the stage, in the system theme (round 3): one quiet line -- "8 of
/// 12 directions · updated 45 s ago", "· catching up" when stale, "Waiting
/// for first views" before the first landing -- and the legend when it fits
/// on one compact line. No sentence of guidance: the ring shows the gaps.
struct WorldCaptureDialCaption: View {
    let dial: WorldCaptureDial
    /// The map the dial draws, or `nil` before the first landing.
    let source: WorldPanelMapSource?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            if let source {
                // The phone's one-second tick, as the map's own label.
                TimelineView(.periodic(from: .now, by: 1)) { _ in
                    line(age: source.receipt.age(at: .now))
                }
            } else {
                line(age: nil)
            }
            WorldCaptureDialLegend()
                .padding(.top, 2)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func line(age: Double?) -> some View {
        Text(WorldCaptureDialCopy.footer(dial, age: age))
            .font(.subheadline.monospacedDigit())
            .foregroundStyle(Color.secondary)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityIdentifier("wb-panel-dial-stamp")
    }
}

/// The dial's legend: one compact line, or nothing. Live and after Stop.
struct WorldCaptureDialLegend: View {
    @Environment(\.colorSchemeContrast) private var contrast
    @ScaledMetric(relativeTo: .caption2) private var swatch: CGFloat = 9

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 12) { legendItems }
                .accessibilityElement(children: .combine)
                .accessibilityIdentifier("wb-panel-dial-legend")
            Color.clear.frame(height: 0)
        }
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
        HStack(alignment: .center, spacing: 4) {
            RoundedRectangle(cornerRadius: 2.5)
                .fill(WorldDialStyle.stage)
                .overlay(RoundedRectangle(cornerRadius: 2.5).fill(fill))
                .overlay(RoundedRectangle(cornerRadius: 2.5)
                    .strokeBorder(edge, style: StrokeStyle(lineWidth: 1, dash: dashed ? [2.5, 1.5] : [])))
                .frame(width: swatch * 1.4, height: swatch)
                .accessibilityHidden(true)
            Text(word)
                .font(.caption2)
                .foregroundStyle(Color.secondary)
                .lineLimit(1)
                .fixedSize()
        }
    }
}

// MARK: - After Stop: the footer

/// After Stop, in the live footer's slot and the system theme: the status
/// line (a small spinner while the Tower works) and the muted line under it
/// -- the panel's headline and the time since Stop -- then the legend. The
/// stage above holds the dimmed dial alone, with no words over it.
struct WorldCaptureDialFinishCaption: View {
    let showsSpinner: Bool
    let status: String
    let headline: String?
    let stoppedAt: ContinuousClock.Instant?
    /// What VoiceOver reads for the status line, given the elapsed text.
    let spoken: (WorldElapsedText?) -> String

    var body: some View {
        if let stoppedAt {
            // Every 30 s: whole minutes, as the finish block's own line.
            TimelineView(.periodic(from: .now, by: 30)) { _ in
                content(WorldElapsedText(since: stoppedAt, now: ContinuousClock.now))
            }
        } else {
            content(nil)
        }
    }

    private func content(_ elapsed: WorldElapsedText?) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .center, spacing: 6) {
                if showsSpinner {
                    ProgressView()
                        .controlSize(.mini)
                        .accessibilityHidden(true)
                }
                Text(status)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(Color.primary)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(spoken(elapsed))
            .accessibilityIdentifier("wb-finish-line")
            Text(WorldCaptureDialCopy.finishStamp(headline: headline, elapsed: elapsed?.visible))
                .font(.footnote.monospacedDigit())
                .foregroundStyle(Color.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .accessibilityIdentifier("wb-panel-dial-stamp")
            WorldCaptureDialLegend()
                .padding(.top, 2)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Recently captured

/// Under the footer line, live and after Stop (round 3): the walk's newest
/// landed photos, newest first, large enough to recognise, in one row that
/// scrolls sideways; a tap opens one large. Nothing when there is none.
struct WorldDialRecentStrip: View {
    let photos: [WorldDialThumbnail]

    static let height: CGFloat = 100

    @State private var selected: WorldDialThumbnail?

    var body: some View {
        if !photos.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                Text(WorldCaptureDialCopy.recentTitle)
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(Color.secondary)
                    .accessibilityAddTraits(.isHeader)
                    .padding(.horizontal, 12)
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(Array(photos.enumerated()), id: \.element.digest) { index, photo in
                            Button { selected = photo } label: {
                                Image(uiImage: photo.image)
                                    .resizable()
                                    .scaledToFill()
                                    .frame(width: Self.height * 16 / 9, height: Self.height)
                                    .clipShape(RoundedRectangle(cornerRadius: 8))
                                    .contentShape(RoundedRectangle(cornerRadius: 8))
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel(WorldCaptureDialCopy.recentPhoto(index + 1, of: photos.count))
                            .accessibilityIdentifier("wb-panel-dial-recent")
                        }
                    }
                    .padding(.horizontal, 12)
                }
            }
            .padding(.bottom, 10)
            // An open sheet never outlives its photo.
            .onChange(of: photos) { _, photos in
                if let shown = selected, !photos.contains(where: { $0.digest == shown.digest }) { selected = nil }
            }
            .sheet(item: $selected) { photo in
                WorldCaptureDialPhotoSheet(photo: photo)
            }
        }
    }
}

/// A tapped photo, large: the image at the sheet's full width and a close
/// button. Nothing else -- no direction, no claim.
struct WorldCaptureDialPhotoSheet: View {
    let photo: WorldDialThumbnail

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                Image(uiImage: photo.image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .accessibilityLabel(WorldImageryCopy.photoNoun(unredacted: photo.unredacted))
                    .accessibilityIdentifier("wb-panel-dial-photo-sheet")
                Spacer(minLength: 0)
            }
            .padding()
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(WorldCaptureDialCopy.closePhoto) { dismiss() }
                        .accessibilityIdentifier("wb-panel-dial-photo-close")
                }
            }
        }
        .presentationDetents([.medium, .large])
    }
}
