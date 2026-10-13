//
//  StatusPill.swift
//  Glasses
//

import SwiftUI

/// The health of one stage of the pipeline.
///
/// Every case pairs a tint with a distinct SF Symbol so status is never
/// communicated by colour alone.
enum StatusLevel {
    case ok
    case working
    case idle
    case problem

    var tint: Color {
        switch self {
        case .ok: return .green
        case .working: return .orange
        case .idle: return .secondary
        case .problem: return .red
        }
    }

    var symbol: String {
        switch self {
        case .ok: return "checkmark.circle.fill"
        case .working: return "clock.fill"
        case .idle: return "circle.dashed"
        case .problem: return "exclamationmark.triangle.fill"
        }
    }
}

/// One stage of the Glasses → Phone → Tower pipeline.
struct StatusPill: View {
    /// How the pill is laid out. `.column` is the glanceable one: symbol over
    /// title over value, three across. `.row` gives the value the whole width
    /// of the screen, for the accessibility text sizes, where a third of the
    /// width broke "Connected" into "Con- nected" and "unavailable" into
    /// "unavail- able" (UX audit, shell-status-bar).
    enum Layout {
        case column
        case row
    }

    let title: String
    let value: String
    let level: StatusLevel
    var layout: Layout = .column

    @Environment(\.accessibilityDifferentiateWithoutColor) private var differentiateWithoutColor

    var body: some View {
        content
            // Paired with the grouped page background so the pill reads as a
            // raised card in light mode and an elevated surface in dark mode.
            .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 14))
            .accessibilityElement(children: .combine)
            .accessibilityLabel(title)
            .accessibilityValue(value)
    }

    @ViewBuilder
    private var content: some View {
        switch layout {
        case .column:
            VStack(spacing: 6) {
                symbol
                    .font(.title3)

                Text(title)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.readableSecondary)

                Text(value)
                    .font(.footnote.weight(.semibold))
                    .multilineTextAlignment(.center)
                    .minimumScaleFactor(0.8)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
            .padding(.horizontal, 8)
        case .row:
            // The symbol beside the title rather than beside the value, so
            // the value -- the words that were breaking -- has the full width.
            VStack(alignment: .leading, spacing: 4) {
                Label {
                    Text(title)
                        .foregroundStyle(.readableSecondary)
                } icon: {
                    symbol
                }
                .font(.caption.weight(.medium))

                Text(value)
                    .font(.footnote.weight(.semibold))
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, 10)
            .padding(.horizontal, 14)
        }
    }

    private var symbol: some View {
        Image(systemName: level.symbol)
            .foregroundStyle(differentiateWithoutColor ? Color.primary : level.tint)
            .symbolRenderingMode(.hierarchical)
    }
}

#Preview {
    HStack(spacing: 10) {
        StatusPill(title: "Glasses", value: "Registered", level: .ok)
        StatusPill(title: "Camera", value: "Streaming", level: .ok)
        StatusPill(title: "Tower", value: "Offline", level: .idle)
    }
    .padding()
}
