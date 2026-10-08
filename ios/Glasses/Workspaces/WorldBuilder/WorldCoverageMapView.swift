//
//  WorldCoverageMapView.swift
//  Glasses
//
//  A fog field in the panel's map slot: each component is a separate piece,
//  with lit directions fading into unseen space. The caption says how old
//  the solve is.
//

import SwiftUI

/// What the map slot draws: the Tower's receipt, or the DEBUG fixture.
enum WorldPanelMapSource: Equatable {
    case coverage(WorldCoverageReceipt)
    case fixture(WorldCoverageReceipt)

    var receipt: WorldCoverageReceipt {
        switch self {
        case .coverage(let receipt), .fixture(let receipt): return receipt
        }
    }

    var isFixture: Bool {
        if case .fixture = self { return true }
        return false
    }
}

enum WorldCoverageStyle {
    /// The entire piece is this slate until a direction lights it.
    static let fog = WorldChromeStyle.rgb(0x182936)
    static let lit = WorldChromeStyle.rgb(0x6DF5DD)
    static let station = WorldChromeStyle.rgb(0xD6FFF5)

    static let supportedGradient = Gradient(stops: [
        .init(color: lit.opacity(WorldCoverageField.supportedIntensity), location: 0),
        .init(color: lit.opacity(0.46), location: 0.3),
        .init(color: lit.opacity(0.24), location: 0.68),
        .init(color: lit.opacity(0), location: 1),
    ])
    static let weakGradient = Gradient(stops: [
        .init(color: lit.opacity(WorldCoverageField.weakIntensity), location: 0),
        .init(color: lit.opacity(0.22), location: 0.3),
        .init(color: lit.opacity(0.09), location: 0.7),
        .init(color: lit.opacity(0), location: 1),
    ])
    static let stationGradient = Gradient(colors: [station.opacity(0.55), lit.opacity(0.15), lit.opacity(0)])
}

/// Solve evidence in screen-independent cell coordinates. Sector zero is
/// screen-right; sector numbers advance clockwise on the y-down map.
struct WorldCoverageField {
    struct Wedge {
        let center: CGPoint
        let sector: Int
        let startAngle: Double
        let endAngle: Double
        let radiusInCells: CGFloat
        let intensity: Double
        let evidence: WorldCoverageEvidence
    }

    static let supportedIntensity = 0.62
    static let weakIntensity = 0.34
    static let supportedRadius: CGFloat = 2.5
    static let weakRadius: CGFloat = 1.6
    static let blurRadiusInCells: CGFloat = 0.28

    /// The eight-cell grid and its largest possible wedge share one fit.
    /// A centred scale of an eight-cell overlay has exactly this mapping.
    struct Fit {
        let cell: CGFloat
        let inset: CGFloat
        let overlayScale: CGFloat

        func point(_ gridPoint: CGPoint) -> CGPoint {
            CGPoint(x: inset + gridPoint.x * cell, y: inset + gridPoint.y * cell)
        }
    }

    static func fit(side: CGFloat) -> Fit {
        let paddedGrid = CGFloat(WorldCoverage.gridSide) + 2 * supportedRadius
        let cell = side / paddedGrid
        return Fit(cell: cell, inset: supportedRadius * cell,
                   overlayScale: CGFloat(WorldCoverage.gridSide) / paddedGrid)
    }

    let stations: [CGPoint]
    let wedges: [Wedge]
    let supportedDirections: Int
    let weakDirections: Int

    var stationCount: Int { stations.count }
    var seenDirections: Int { supportedDirections + weakDirections }
    var totalDirections: Int { stationCount * WorldCoverage.sectors }
    var accessibilitySummary: String {
        Self.summary(seen: seenDirections, total: totalDirections, places: stationCount)
    }

    init(component: WorldCoverage.Component) {
        var stations: [CGPoint] = []
        var wedges: [Wedge] = []
        stations.reserveCapacity(component.stations.count)
        wedges.reserveCapacity(component.stations.count * WorldCoverage.sectors)
        var supported = 0
        var weak = 0
        for station in component.stations {
            let center = CGPoint(x: CGFloat(station.x) + 0.5, y: CGFloat(station.y) + 0.5)
            stations.append(center)
            for sector in 0..<WorldCoverage.sectors {
                let evidence = station.evidence(sector: sector)
                let radius: CGFloat
                let intensity: Double
                switch evidence {
                case .supported:
                    supported += 1
                    radius = Self.supportedRadius
                    intensity = Self.supportedIntensity
                case .weak:
                    weak += 1
                    radius = Self.weakRadius
                    intensity = Self.weakIntensity
                case .unconfirmed:
                    continue
                }
                let span = WorldCoverageGeometry.span(sector: sector)
                wedges.append(Wedge(center: center, sector: sector, startAngle: span.start,
                                    endAngle: span.end, radiusInCells: radius, intensity: intensity,
                                    evidence: evidence))
            }
        }
        self.stations = stations
        self.wedges = wedges
        self.supportedDirections = supported
        self.weakDirections = weak
    }

    static func summary(for components: [WorldCoverage.Component]) -> String {
        var seen = 0, total = 0, places = 0
        for component in components {
            let field = WorldCoverageField(component: component)
            seen += field.seenDirections
            total += field.totalDirections
            places += field.stationCount
        }
        return summary(seen: seen, total: total, places: places)
    }

    private static func summary(seen: Int, total: Int, places: Int) -> String {
        "Seen in \(seen) of \(total) directions at \(places) \(places == 1 ? "place" : "places")"
    }
}

/// The stage's drawing: the tray of pieces. Frozen (after Stop) it is a
/// backdrop under the finish block.
struct WorldCoverageMapView: View {
    let coverage: WorldCoverage
    let isFrozen: Bool
    /// FOW v1.1 B2: landed thumbnails over this receipt's pieces, or `nil`
    /// (the switch off, or none to draw): the map exactly as v1 draws it.
    var landed: WorldLandedOverlay? = nil

    @State private var selected: WorldLandedThumbnail?

    var body: some View {
        GeometryReader { proxy in
            let pieces = coverage.components
            let named = pieces.count > 1
            let frames = WorldCoverageGeometry.pieceFrames(
                count: pieces.count, in: CGSize(width: proxy.size.width, height: proxy.size.height))
            ZStack(alignment: .topLeading) {
                ForEach(Array(pieces.enumerated()), id: \.element.referenceSegment) { index, component in
                    let frame = frames[index]
                    WorldCoveragePieceView(component: component, index: index, showsName: named && !isFrozen)
                        .frame(width: frame.width, height: frame.height)
                        .position(x: frame.midX, y: frame.midY)
                    // Over the piece, never inside its element: the v1
                    // piece's own label and value are unchanged.
                    if let landed, !isFrozen, case let thumbnails = landed.thumbnails(on: component.referenceSegment),
                       !thumbnails.isEmpty {
                        WorldLandedPieceOverlay(thumbnails: thumbnails, receipt: landed.receipt) { selected = $0 }
                            .frame(width: frame.width, height: frame.height)
                            .scaleEffect(WorldCoverageField.fit(side: frame.width).overlayScale)
                            .position(x: frame.midX, y: frame.midY)
                    }
                }
            }
        }
        // An open sheet never outlives its thumbnail: a newer landing, a
        // switch going off or another walk's map closes it.
        .onChange(of: landed) { _, overlay in
            if let shown = selected, overlay?.thumbnails.contains(shown) != true { selected = nil }
        }
        .sheet(item: $selected) { thumbnail in
            WorldLandedThumbnailSheet(thumbnail: thumbnail, receipt: landed?.receipt)
        }
        .opacity(isFrozen ? 0.35 : 1)
        // The map's own element is a leaf under the pieces, read first: a
        // container that is the stage's only child is folded into the
        // stage by SwiftUI, and its identity lost.
        .background {
            Rectangle()
                .fill(Color.white.opacity(0.001))
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("\(WorldCoverageCopy.mapName). \(WorldCoverageField.summary(for: coverage.components))")
                .accessibilityValue(WorldCoverageCopy.mapValue(coverage))
                .accessibilityIdentifier("wb-panel-map")
                .accessibilitySortPriority(1)
        }
    }
}

/// One component: fog everywhere, with evidence lighting only seen headings.
struct WorldCoveragePieceView: View {
    let component: WorldCoverage.Component
    let index: Int
    let showsName: Bool

    var body: some View {
        let field = WorldCoverageField(component: component)
        Canvas { context, size in
            let side = min(size.width, size.height)
            let fit = WorldCoverageField.fit(side: side)
            let cell = fit.cell
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(WorldCoverageStyle.fog))
            context.drawLayer { revealed in
                revealed.addFilter(.blur(radius: cell * WorldCoverageField.blurRadiusInCells))
                for wedge in field.wedges {
                    let center = fit.point(wedge.center)
                    let radius = wedge.radiusInCells * cell
                    var path = Path()
                    path.move(to: center)
                    path.addArc(center: center, radius: radius, startAngle: .degrees(wedge.startAngle),
                                endAngle: .degrees(wedge.endAngle), clockwise: false)
                    path.closeSubpath()
                    let gradient = wedge.evidence == .supported
                        ? WorldCoverageStyle.supportedGradient : WorldCoverageStyle.weakGradient
                    revealed.fill(path, with: .radialGradient(gradient, center: center,
                                                              startRadius: 0, endRadius: radius))
                }
            }
            for station in field.stations {
                let center = fit.point(station)
                let radius = max(1.5, cell * 0.13)
                let rect = CGRect(x: center.x - radius, y: center.y - radius,
                                  width: radius * 2, height: radius * 2)
                context.fill(Path(ellipseIn: rect), with: .radialGradient(WorldCoverageStyle.stationGradient,
                                                                          center: center, startRadius: 0,
                                                                          endRadius: radius))
            }
        }
        .overlay(alignment: .topLeading) {
            if showsName {
                Text(WorldCoverageCopy.pieceName(index))
                    .accessibilityHidden(true)
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(WorldChromeStyle.text)
                    .padding(.horizontal, 4)
                    .background(WorldChromeStyle.pill, in: RoundedRectangle(cornerRadius: 4))
                    .padding(2)
                    .dynamicTypeSize(...DynamicTypeSize.xxxLarge)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(WorldCoverageCopy.pieceName(index))
        .accessibilityValue(WorldCoverageCopy.pieceValue(component))
        .accessibilityIdentifier("wb-panel-map-piece")
    }
}

/// The map's words, under the stage while walking and inside the finish
/// block after Stop: the dated label, the newer keyframes, the pieces and
/// what the caps left out, and (walking) the legend. The age is refreshed on
/// the phone each second from the receipt's clock; the Tower is never asked.
struct WorldCoverageCaption: View {
    let source: WorldPanelMapSource
    /// After Stop: the frozen map's wording, on the dark plate, no legend.
    let stopped: Bool

    private var receipt: WorldCoverageReceipt { source.receipt }
    private var primary: Color { stopped ? WorldChromeStyle.text : Color.primary }
    private var secondary: Color { stopped ? WorldChromeStyle.secondary : Color.primary }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            TimelineView(.periodic(from: .now, by: 1)) { _ in
                line(WorldCoverageCopy.label(receipt, at: .now, stopped: stopped), "wb-panel-map-label",
                     font: .footnote.weight(.semibold), color: primary)
            }
            if let newer = WorldCoverageCopy.newer(receipt, stopped: stopped) {
                line(newer, "wb-panel-map-newer", font: .footnote, color: secondary)
            }
            if !stopped, let pieces = WorldCoverageCopy.pieces(receipt.coverage) {
                line(pieces, "wb-panel-map-pieces", font: .footnote, color: secondary)
            }
            if let notDrawn = WorldCoverageCopy.notDrawn(receipt.coverage) {
                line(notDrawn, "wb-panel-map-not-drawn", font: .footnote, color: secondary)
            }
            if !stopped {
                legend
            }
            if source.isFixture {
                line(WorldCoverageCopy.fixture, "wb-panel-map-fixture", font: .footnote, color: secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
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
        VStack(alignment: .leading, spacing: 2) {
            legendRow("Seen", color: WorldCoverageStyle.lit)
            legendRow("Not yet seen", color: WorldCoverageStyle.fog)
        }
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("wb-panel-map-legend")
    }

    private func legendRow(_ word: String, color: Color) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            RoundedRectangle(cornerRadius: 2)
                .fill(color)
                .frame(width: 10, height: 10)
                .accessibilityHidden(true)
            Text(word)
                .font(.footnote)
                .foregroundStyle(Color.primary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    static func legendWord(_ evidence: WorldCoverageEvidence) -> String {
        switch evidence {
        case .unconfirmed: return WorldCoverageCopy.unconfirmed
        case .weak: return WorldCoverageCopy.weak
        case .supported: return WorldCoverageCopy.supported
        }
    }
}
