//
//  WorldCoverageMapView.swift
//  Glasses
//
//  Fog of war v1 in the panel's map slot (U-INLINE S1/S2): one top-down
//  station grid per component, each station a fan of 12 heading sectors --
//  grey, muted or colored -- and separate components as separate pieces in
//  a tray, never merged. The caption under it says how old the map is.
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
    /// Not yet seen by a finished solve: grey, never "a hole".
    static let unconfirmed = Color(white: 0.34)
    /// One view: muted, and not covered.
    static let weak = WorldChromeStyle.rgb(0x2E6B66)
    /// Two or more views.
    static let supported = WorldChromeStyle.rgb(0x4FD8C8)
    static let cell = Color.white.opacity(0.05)
    static let cellLine = Color.white.opacity(0.10)

    static func color(_ evidence: WorldCoverageEvidence) -> Color {
        switch evidence {
        case .unconfirmed: return unconfirmed
        case .weak: return weak
        case .supported: return supported
        }
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
                            .position(x: frame.midX, y: frame.midY)
                    }
                }
            }
        }
        .sheet(item: $selected) { thumbnail in
            if let landed {
                WorldLandedThumbnailSheet(thumbnail: thumbnail, receipt: landed.receipt)
            }
        }
        .opacity(isFrozen ? 0.35 : 1)
        // The map's own element is a leaf under the pieces, read first: a
        // container that is the stage's only child is folded into the
        // stage by SwiftUI, and its identity lost.
        .background {
            Rectangle()
                .fill(Color.white.opacity(0.001))
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(WorldCoverageCopy.mapName)
                .accessibilityValue(WorldCoverageCopy.mapValue(coverage))
                .accessibilityIdentifier("wb-panel-map")
                .accessibilitySortPriority(1)
        }
    }
}

/// One component: its 8 x 8 station grid and one fan per station.
struct WorldCoveragePieceView: View {
    let component: WorldCoverage.Component
    let index: Int
    let showsName: Bool

    var body: some View {
        Canvas { context, size in
            let side = min(size.width, size.height)
            let cell = side / CGFloat(WorldCoverage.gridSide)
            for row in 0..<WorldCoverage.gridSide {
                for column in 0..<WorldCoverage.gridSide {
                    let rect = CGRect(x: CGFloat(column) * cell, y: CGFloat(row) * cell, width: cell, height: cell)
                        .insetBy(dx: 0.5, dy: 0.5)
                    context.fill(Path(rect), with: .color(WorldCoverageStyle.cell))
                    context.stroke(Path(rect), with: .color(WorldCoverageStyle.cellLine), lineWidth: 0.5)
                }
            }
            for fan in WorldCoverageGeometry.fans(component, side: side) {
                for (sector, evidence) in fan.sectors.enumerated() {
                    let span = WorldCoverageGeometry.span(sector: sector)
                    var path = Path()
                    path.move(to: fan.centre)
                    path.addArc(center: fan.centre, radius: fan.radius, startAngle: .degrees(span.start + 1.5),
                                endAngle: .degrees(span.end - 1.5), clockwise: false)
                    path.closeSubpath()
                    context.fill(path, with: .color(WorldCoverageStyle.color(evidence)))
                }
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
            ForEach(WorldCoverageEvidence.allCases, id: \.self) { evidence in
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(WorldCoverageStyle.color(evidence))
                        .frame(width: 10, height: 10)
                        .accessibilityHidden(true)
                    Text(Self.legendWord(evidence))
                        .font(.footnote)
                        .foregroundStyle(Color.primary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("wb-panel-map-legend")
    }

    static func legendWord(_ evidence: WorldCoverageEvidence) -> String {
        switch evidence {
        case .unconfirmed: return WorldCoverageCopy.unconfirmed
        case .weak: return WorldCoverageCopy.weak
        case .supported: return WorldCoverageCopy.supported
        }
    }
}
