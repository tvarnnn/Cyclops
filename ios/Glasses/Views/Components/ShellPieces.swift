//
//  ShellPieces.swift
//  Glasses
//

import SwiftUI

/// Small views shared by the shell and by every workspace.
///
/// These were `private` inside `SessionView`. That file is gone — its status
/// row became the persistent shell, its setup rows moved behind
/// `ConnectionSheet`, and its body became `HomeWorkspaceView` — and these
/// pieces are now needed in more than one of those places. Promoting them is
/// what stops each workspace from growing its own slightly different section
/// heading.

/// A quiet uppercase heading above a group.
struct SectionLabel: View {
    let text: String
    init(_ text: String) { self.text = text }

    var body: some View {
        Text(text.uppercased())
            .font(.caption2.weight(.semibold))
            .foregroundStyle(.readableSecondary)
            .tracking(0.6)
            .padding(.leading, 4)
            // A heading to VoiceOver too, so the rotor can move between the
            // groups it heads.
            .accessibilityAddTraits(.isHeader)
    }
}

/// Secondary explanatory text under a control.
struct HelperText: View {
    let text: String
    init(_ text: String) { self.text = text }

    var body: some View {
        Text(text)
            .font(.caption)
            .foregroundStyle(.readableSecondary)
            .multilineTextAlignment(.center)
            .frame(maxWidth: .infinity)
    }
}

/// One step of setup, with its action always reachable.
///
/// Completion is shown as a leading checkmark rather than by replacing the
/// action, so the underlying call stays available in every state — the property
/// that keeps a manual Tower reconnect reachable after the automatic schedule
/// has given up.
struct SetupRow: View {
    let title: String
    let detail: String
    let isComplete: Bool
    let actionTitle: String
    let action: () -> Void

    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    var body: some View {
        HStack(alignment: dynamicTypeSize.isAccessibilitySize ? .top : .center, spacing: 10) {
            Image(systemName: isComplete ? "checkmark.circle.fill" : "circle.dashed")
                .foregroundStyle(isComplete ? Color.green : Color.secondary)
                .accessibilityHidden(true)

            // The action under the words at the accessibility sizes: a
            // trailing column left the detail so narrow that "unavailable"
            // broke into "unavail- able" (UX audit, connection-sheet).
            if dynamicTypeSize.isAccessibilitySize {
                VStack(alignment: .leading, spacing: 6) {
                    words
                    actionButton
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                words
                Spacer(minLength: 8)
                actionButton
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .accessibilityElement(children: .contain)
        .accessibilityValue(isComplete ? "Done. \(detail)" : detail)
    }

    private var words: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
            Text(detail)
                .font(.caption)
                .foregroundStyle(.readableSecondary)
        }
        .fixedSize(horizontal: false, vertical: true)
    }

    private var actionButton: some View {
        Button(actionTitle, action: action)
            .font(.subheadline.weight(.medium))
            .buttonStyle(.borderless)
            .accessibilityLabel("\(actionTitle) \(title)")
            // Voice Control matches the visible word.
            .accessibilityInputLabels([actionTitle, "\(actionTitle) \(title)"])
    }
}

/// A problem the user can act on, stated in full.
struct FailureBanner: View {
    let text: String
    /// Shown as a trailing control when the failure has an obvious remedy —
    /// or under the text at the accessibility sizes, where a trailing column
    /// is too narrow for a word to fit on a line.
    var actionTitle: String?
    var action: (() -> Void)?

    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            // Decoration: the words say it is a problem.
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
                .accessibilityHidden(true)
            if dynamicTypeSize.isAccessibilitySize {
                VStack(alignment: .leading, spacing: 8) {
                    message
                    actionButton
                }
            } else {
                message
                Spacer(minLength: 0)
                actionButton
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 12))
        .accessibilityElement(children: .contain)
    }

    private var message: some View {
        Text(text)
            .font(.footnote)
            .fixedSize(horizontal: false, vertical: true)
    }

    @ViewBuilder
    private var actionButton: some View {
        if let actionTitle, let action {
            Button(actionTitle, action: action)
                .font(.footnote.weight(.medium))
                .buttonStyle(.borderless)
                // Footnote-sized tinted text on a card: the system blue was
                // just under 4.5:1 there ("Tower settings", on Home).
                .tint(Color.readableTint)
        }
    }
}
