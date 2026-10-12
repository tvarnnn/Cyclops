//
//  WorldPhotoFloorplanView.swift
//  Glasses
//
//  FOW v1.2: one piece of the photo floor plan -- a neutral gray plate with
//  the decoded RGBA raster over it, fitted by the manifest's padded bounds
//  in the piece's own basis. Alpha 0 lets the gray show through: "No usable
//  photo here yet". No fans, no station dots on the photo.
//

import SwiftUI

enum WorldPhotoFloorplanStyle {
    /// The plate, and every alpha-0 cell of the raster.
    static let gray = WorldChromeStyle.rgb(0x70757B)
}

enum WorldPhotoFloorplanCopy {
    static let gray = "No usable photo here yet"
    static let updating = "Updating…"
    static let notImproved = "Couldn't improve this view yet"

    /// The caption under a held plan, and its identifier.
    static func held(_ held: WorldPhotoDisplay.Held) -> (words: String, identifier: String) {
        switch held {
        case .updating: return (updating, "wb-panel-map-updating")
        case .notImproved: return (notImproved, "wb-panel-map-not-improved")
        }
    }

    static func pieceValue(_ piece: WorldPhotoPiece) -> String {
        let views = piece.projectedKeyframes == 1 ? "1 keyframe" : "\(piece.projectedKeyframes) keyframes"
        return "Photo floor plan from \(views). Gray: \(gray.lowercased())"
    }
}

/// One component as a photo: gray everywhere, the raster where it lies.
struct WorldPhotoPieceView: View {
    let piece: WorldPhotoPiece
    let index: Int
    let showsName: Bool

    var body: some View {
        let image = Image(uiImage: piece.image)
        Canvas { context, size in
            let side = min(size.width, size.height)
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(WorldPhotoFloorplanStyle.gray))
            context.draw(image, in: piece.rasterRect(side: side))
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
        .accessibilityValue(WorldPhotoFloorplanCopy.pieceValue(piece))
        .accessibilityIdentifier("wb-panel-map-photo")
    }
}
