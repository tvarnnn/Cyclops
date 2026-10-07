//
//  WorldImageryViews.swift
//  Glasses
//
//  FOW v1.1 B's two drawings (spec §5): B1's strip, its own card under the
//  current piece and never on the map; B2's thumbnails, layered over the v1
//  grid's pieces, never replacing a fan and never on a grey cell. Every
//  thumbnail is the Tower's redacted tile; each one is labelled for
//  VoiceOver, and none is ever announced.
//

import SwiftUI

enum WorldImageryCopy {
    static let liveLabel = "Current piece · unplaced"
    static let liveEmpty = "No photos of this stretch yet"
    static let liveHonesty = "Redacted photos of the current stretch. Not placed on the map."
    static let liveThumbnail = "current piece photo, unplaced"
    static let landedHonesty = "Redacted photo, placed by the last finished solve. Not a wall or a surface."

    /// "look direction 90°", clockwise from the piece's sector zero.
    static func direction(_ heading: Double?) -> String {
        guard let heading else { return "look direction unknown" }
        return "look direction \(Int(heading.rounded()) % 360)°"
    }

    /// "12 seconds old", "· stale" above 30 s (coverage §8.2), or "age unavailable".
    static func age(_ seconds: Double?) -> String {
        guard let seconds else { return "age unavailable" }
        let whole = Int(seconds.rounded(.down))
        let stale = seconds > WorldCoverageReceipt.staleAfter ? ", stale" : ""
        return (whole == 1 ? "1 second old" : "\(whole) seconds old") + stale
    }

    /// VoiceOver: "redacted photo, look direction N°, M seconds old".
    static func landedThumbnail(heading: Double?, age seconds: Double?) -> String {
        "redacted photo, \(direction(heading)), \(age(seconds))"
    }
}

// MARK: - B1: the live strip

/// "Current piece · unplaced": the current segment's thumbnails, newest
/// leading, scrolled sideways. Its own card, never drawn against the grid.
struct WorldImageryStripView: View {
    let strip: WorldImageryStrip

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(WorldImageryCopy.liveLabel)
                .font(.subheadline.weight(.semibold))
                .accessibilityIdentifier("wb-imagery-live-label")
            if strip.thumbnails.isEmpty {
                Text(WorldImageryCopy.liveEmpty)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .accessibilityIdentifier("wb-imagery-live-empty")
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(strip.thumbnails) { thumbnail in
                            Image(uiImage: thumbnail.image)
                                .resizable()
                                .aspectRatio(16.0 / 9.0, contentMode: .fill)
                                .frame(width: 96, height: 54)
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                                .accessibilityLabel(WorldImageryCopy.liveThumbnail)
                                .accessibilityIdentifier("wb-imagery-live-thumb")
                        }
                    }
                }
            }
            Text(WorldImageryCopy.liveHonesty)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 18))
        .overlay {
            RoundedRectangle(cornerRadius: 18)
                .strokeBorder(WorldCurrentPieceView.tint.opacity(0.6), style: StrokeStyle(lineWidth: 1, dash: [5, 4]))
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("wb-imagery-live")
    }
}

// MARK: - B2: thumbnails over one piece

/// One piece's landed thumbnails, each at its true projected centre (its
/// image kept inside the piece) with a tick for where it looked. Over the
/// piece's own frame, in the fans' basis: column right, row down.
struct WorldLandedPieceOverlay: View {
    let thumbnails: [WorldLandedThumbnail]
    let receipt: WorldCoverageReceipt
    let select: (WorldLandedThumbnail) -> Void

    var body: some View {
        GeometryReader { proxy in
            let side = min(proxy.size.width, proxy.size.height)
            TimelineView(.periodic(from: .now, by: 1)) { _ in
                layer(side: side, age: receipt.age(at: .now))
            }
        }
    }

    private func layer(side: CGFloat, age: Double?) -> some View {
        let cell = side / CGFloat(WorldCoverage.gridSide)
        let size = Self.thumbnailSize(cell: cell)
        return ZStack(alignment: .topLeading) {
            ForEach(thumbnails) { thumbnail in
                WorldLandedThumbnailButton(thumbnail: thumbnail, size: size, age: age) { select(thumbnail) }
                    .position(Self.placed(thumbnail, cell: cell, size: size, side: side))
            }
            WorldLandedTicks(thumbnails: thumbnails, cell: cell)
                .allowsHitTesting(false)
                .accessibilityHidden(true)
        }
        .frame(width: side, height: side, alignment: .topLeading)
    }

    static func thumbnailSize(cell: CGFloat) -> CGSize {
        let width = min(max(cell * 1.6, 28), 64)
        return CGSize(width: width, height: width * 9 / 16)
    }

    /// The true projected centre, in the piece's points.
    static func centre(_ thumbnail: WorldLandedThumbnail, cell: CGFloat) -> CGPoint {
        CGPoint(x: CGFloat(thumbnail.column) * cell, y: CGFloat(thumbnail.row) * cell)
    }

    /// The image's centre: the true centre, moved only as far as keeps the
    /// image inside the piece.
    static func placed(_ thumbnail: WorldLandedThumbnail, cell: CGFloat, size: CGSize, side: CGFloat) -> CGPoint {
        let centre = centre(thumbnail, cell: cell)
        let x = min(max(centre.x, size.width / 2), side - size.width / 2)
        let y = min(max(centre.y, size.height / 2), side - size.height / 2)
        return CGPoint(x: x, y: y)
    }
}

/// One landed thumbnail: tap for where it looked and how old it is.
struct WorldLandedThumbnailButton: View {
    let thumbnail: WorldLandedThumbnail
    let size: CGSize
    let age: Double?
    let tap: () -> Void

    var body: some View {
        Button(action: tap) {
            Image(uiImage: thumbnail.image)
                .resizable()
                .aspectRatio(16.0 / 9.0, contentMode: .fill)
                .frame(width: size.width, height: size.height)
                .clipShape(RoundedRectangle(cornerRadius: 3))
                .overlay(RoundedRectangle(cornerRadius: 3).strokeBorder(Color.white.opacity(0.8), lineWidth: 1))
        }
        .buttonStyle(.plain)
        .accessibilityLabel(WorldImageryCopy.landedThumbnail(heading: thumbnail.heading, age: age))
        .accessibilityIdentifier("wb-imagery-landed-thumb")
    }
}

/// A dot at each true centre and a tick the way it looked: clockwise from
/// screen-right, as the fans' sectors run.
struct WorldLandedTicks: View {
    let thumbnails: [WorldLandedThumbnail]
    let cell: CGFloat

    var body: some View {
        Canvas { context, _ in
            for thumbnail in thumbnails {
                let centre = WorldLandedPieceOverlay.centre(thumbnail, cell: cell)
                let dot = CGRect(x: centre.x - 2.5, y: centre.y - 2.5, width: 5, height: 5)
                context.fill(Path(ellipseIn: dot), with: .color(.white))
                guard let heading = thumbnail.heading else { continue }
                let radians = heading * .pi / 180
                let length = Double(cell) * 0.7
                var tick = Path()
                tick.move(to: centre)
                tick.addLine(to: CGPoint(x: Double(centre.x) + cos(radians) * length,
                                         y: Double(centre.y) + sin(radians) * length))
                context.stroke(tick, with: .color(.white), lineWidth: 2)
            }
        }
    }
}

/// A tapped thumbnail: the photo, where it looked, and how old the solve is.
struct WorldLandedThumbnailSheet: View {
    let thumbnail: WorldLandedThumbnail
    /// The receipt the thumbnail was placed on; `nil` reads "age unavailable".
    let receipt: WorldCoverageReceipt?

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                Image(uiImage: thumbnail.image)
                    .resizable()
                    .aspectRatio(16.0 / 9.0, contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .accessibilityLabel("redacted photo")
                Text(WorldImageryCopy.direction(thumbnail.heading).capitalizedFirst)
                    .font(.headline)
                    .accessibilityIdentifier("wb-imagery-landed-direction")
                TimelineView(.periodic(from: .now, by: 1)) { _ in
                    Text(WorldImageryCopy.age(receipt?.age(at: .now)).capitalizedFirst)
                        .font(.subheadline)
                        .accessibilityIdentifier("wb-imagery-landed-age")
                }
                Text(WorldImageryCopy.landedHonesty)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 0)
            }
            .padding()
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium, .large])
    }
}

private extension String {
    var capitalizedFirst: String { prefix(1).uppercased() + dropFirst() }
}
