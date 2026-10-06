//
//  WorldFinishViews.swift
//  Glasses
//
//  U0.6: the finish block (spinner, stage line, elapsed line) and the
//  foreground banner. The words are `WorldFinishProgress.swift`'s.
//

import SwiftUI

/// The finish block: an optional spinner, the stage line, the elapsed line.
/// One accessibility element, so VoiceOver reads it as one sentence.
struct WorldFinishLineView: View {
    enum Style: Equatable {
        /// The World Builder panel's stage area: the dark plate, large text.
        case panel
        /// The 3D viewer's caption.
        case viewer
        /// The 3D viewer's native top band (U1.1): its own colours.
        case band
    }

    let showsSpinner: Bool
    let line: WorldFinishLine?
    let stoppedAt: ContinuousClock.Instant?
    let style: Style

    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    var body: some View {
        if showsSpinner || line != nil || stoppedAt != nil {
            if let stoppedAt {
                // Every 30 s: whole minutes, so nothing faster is ever true.
                TimelineView(.periodic(from: .now, by: 30)) { _ in
                    content(elapsed: WorldElapsedText(since: stoppedAt, now: ContinuousClock.now))
                }
            } else {
                content(elapsed: nil)
            }
        }
    }

    private func content(elapsed: WorldElapsedText?) -> some View {
        VStack(alignment: .leading, spacing: style == .panel ? 8 : 4) {
            if showsSpinner || line != nil {
                let layout = dynamicTypeSize.isAccessibilitySize
                    ? AnyLayout(VStackLayout(alignment: .leading, spacing: 6))
                    : AnyLayout(HStackLayout(alignment: .firstTextBaseline, spacing: 10))
                layout {
                    if showsSpinner {
                        ProgressView()
                            .tint(style == .viewer ? nil : WorldChromeStyle.text)
                    }
                    Text(line?.text ?? WorldFinishCopy.finishing)
                        .font(lineFont)
                        .foregroundStyle(primary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            if let elapsed {
                Text(elapsed.visible)
                    .font(elapsedFont)
                    .monospacedDigit()
                    .foregroundStyle(secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(WorldFinishCopy.spoken(showsSpinner: showsSpinner, line: line, elapsed: elapsed))
        .accessibilityIdentifier(style == .panel ? "wb-finish-line" : "world-render-finish-line")
    }

    private var lineFont: Font {
        switch style {
        case .panel: return .title3
        case .viewer, .band: return .caption
        }
    }

    private var elapsedFont: Font {
        switch style {
        case .panel: return .body
        case .viewer, .band: return .caption2
        }
    }

    private var primary: AnyShapeStyle {
        switch style {
        case .panel: return AnyShapeStyle(WorldChromeStyle.text)
        case .band: return AnyShapeStyle(WorldChromeStyle.text)
        case .viewer: return AnyShapeStyle(.readableSecondary)
        }
    }

    private var secondary: AnyShapeStyle {
        switch style {
        case .panel, .band: return AnyShapeStyle(WorldChromeStyle.secondary)
        case .viewer: return AnyShapeStyle(.readableSecondary)
        }
    }
}

/// "While you were away, your walk finished: Saved." One OK; never
/// auto-dismissed; announced once when it appears. No haptic: the look-back
/// cue owns that channel.
struct WorldFinishBannerView: View {
    let text: String
    let dismiss: () -> Void

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                label
                Spacer(minLength: 0)
                okButton
            }
            VStack(alignment: .leading, spacing: 10) {
                label
                okButton
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 14))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("wb-finish-banner")
        .onAppear { AccessibilityNotification.Announcement(text).post() }
    }

    private var label: some View {
        Label(text, systemImage: "checkmark.circle")
            .font(.body)
            .fixedSize(horizontal: false, vertical: true)
    }

    private var okButton: some View {
        Button("OK", action: dismiss)
            .readableBorderedButton()
            .frame(minWidth: 44, minHeight: 44)
            .accessibilityHint("Dismisses this message.")
            .accessibilityIdentifier("wb-finish-banner-ok")
    }
}
