//
//  MetricTile.swift
//  Glasses
//

import SwiftUI

/// A single measured value. Only ever fed from a real model property — the
/// dashboard has no computed or estimated metrics.
struct MetricTile: View {
    let caption: String
    let value: String
    /// Optional clarifier shown under the value, for numbers whose meaning is
    /// easy to misread (e.g. frames captured vs. frames actually sent).
    var footnote: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(caption)
                .font(.caption)
                .foregroundStyle(.readableSecondary)

            Text(value)
                .font(.title2.weight(.semibold))
                .monospacedDigit()
                .contentTransition(.numericText())
                .lineLimit(1)
                .minimumScaleFactor(0.6)

            if let footnote {
                Text(footnote)
                    .font(.caption2)
                    .foregroundStyle(.readableSecondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 12))
        .accessibilityElement(children: .combine)
        .accessibilityLabel(caption)
        .accessibilityValue(footnote.map { "\(value), \($0)" } ?? value)
    }
}

#Preview {
    LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
        MetricTile(caption: "Frames from glasses", value: "4,271")
        MetricTile(caption: "Sent to Tower", value: "11.9", footnote: "frames per second")
    }
    .padding()
}
