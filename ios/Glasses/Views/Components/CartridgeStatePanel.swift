//
//  CartridgeStatePanel.swift
//  Glasses
//

import SwiftUI

/// The panel every cartridge workspace uses to say that the Tower has nothing
/// for it.
///
/// ## Why this is shared and the workspaces are not
///
/// Four workspaces need to render "there is nothing here yet", and it is the
/// only screen state any of them can reach today. Four hand-written versions
/// would drift into four different explanations of the same single fact — and
/// the fact is *shared*: the Tower has no module runtime, so no cartridge has a
/// contract. One panel, one wording, one place to change when that stops being
/// true.
///
/// What is emphatically **not** shared is the workspace itself. A generic
/// cartridge screen driven by descriptor metadata is the plugin framework this
/// work is forbidden to build; each workspace stays its own file with its own
/// layout, and borrows this panel the way it borrows `SectionLabel`.
///
/// ## What it refuses to draw
///
/// - **No metric tiles.** Placeholder values render as "—", and a grid of
///   dashes reads as *broken* rather than as *early*. Prose carries the same
///   information and creates no dead UI to explain away later.
/// - **No spinner** unless the phase is `.waiting`. A progress indicator claims
///   work is underway; in `.unsupported` nothing is underway and never will be
///   without a Tower change.
/// - **No "coming soon".** It states what exists now and what would have to
///   change, without promising a date the roadmap does not contain.
struct CartridgeStatePanel: View {
    /// The heading — what section of the workspace this is.
    let title: String
    /// The phase, which decides the icon and whether progress is honest.
    let phase: CartridgePhase
    /// The situation, in a sentence, from
    /// `CartridgeAvailability.explanation(cartridgeName:)` or a failure's
    /// message. Never composed here: this view does not know enough to write
    /// it, and a view that guesses at an explanation is how "not built yet"
    /// becomes "something went wrong".
    let explanation: String
    /// What this panel will show when the contract exists. Optional, and
    /// deliberately phrased as a conditional rather than a promise.
    var futureDescription: String?

    /// Connect and Settings, from the root (U0.8 F03). `nil` in previews.
    @Environment(\.towerRecovery) private var recovery

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel(title)

            VStack(alignment: .leading, spacing: 12) {
                if phase.showsProgress {
                    // The one honest use of a progress indicator: something is
                    // genuinely in flight.
                    HStack(spacing: 10) {
                        ProgressView()
                        Text(explanation)
                            .font(.subheadline)
                            .foregroundStyle(.readableSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                } else {
                    Label(headline, systemImage: symbol)
                        .font(.headline)
                    Text(explanation)
                        .font(.subheadline)
                        .foregroundStyle(.readableSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                if let futureDescription {
                    Text(futureDescription)
                        .font(.footnote)
                        .foregroundStyle(.readableSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 18))
            .accessibilityElement(children: .combine)

            // Outside the combined card, so each is its own control. "Not
            // connected" is the one state a person can end, so it names how.
            if phase == .disconnected, let recovery {
                TowerRecoveryButtons(actions: recovery)
            }
        }
    }

    private var headline: String {
        switch phase {
        case .unsupported: return "Nothing yet"
        // Different words and a different glyph from `.unsupported`, because
        // they are different situations calling for opposite responses — one
        // waits for a Tower that may never do this, the other waits for a
        // connection. The explanation string always distinguished them; the
        // headline and the icon are what a person actually reads first, and
        // they used to say the same thing for both.
        case .disconnected: return "Not connected"
        // Names the action, because there is one. The other empty states are
        // all "wait"; this is the only one a person can end themselves.
        case .needsUpdate: return "Update needed"
        case .idle: return "Not started"
        case .waiting: return "Waiting"
        case .live: return "Working"
        case .settled: return "Ready"
        case .failed: return "Failed"
        }
    }

    private var symbol: String {
        switch phase {
        // Deliberately the same outline glyph for `.unsupported` and `.idle`.
        // A distinct icon per empty state suggests they differ in a way the
        // user can act on, and those two do not.
        case .unsupported, .idle: return "circle.dashed"
        case .disconnected: return "wifi.slash"
        case .needsUpdate: return "arrow.down.circle"
        case .waiting: return "clock"
        case .live: return "dot.radiowaves.left.and.right"
        case .settled: return "checkmark.circle"
        case .failed: return "exclamationmark.triangle.fill"
        }
    }
}

#Preview("Unsupported") {
    CartridgeStatePanel(
        title: "What the Tower knows",
        phase: .unsupported,
        explanation: CartridgeAvailability.noContract.explanation(cartridgeName: "Document Memory") ?? "",
        futureDescription: "When the contract exists, this panel will list documents the Tower observed."
    )
    .padding()
}
