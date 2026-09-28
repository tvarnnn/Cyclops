//
//  ViewfinderCard.swift
//  Glasses
//

// `CapturedFrame` and the whole camera path are DEBUG-only in the model
// (Glasses/GlassesConnection.swift). This entire file is gated to match, so no
// control ever crosses the conditional boundary.
#if DEBUG

import SwiftUI

/// Shows the most recent frame decoded from the glasses.
///
/// This renders `GlassesConnection.latestCapturedFrame`, which the model
/// samples down to `FrameRateGate.towerTargetFPS` — the same frames that go to
/// the Tower, so what you see is what was sent. It deliberately does not open
/// its own camera feed or change that cadence.
///
/// A new `UIImage` identity per frame defeats SwiftUI's image caching, so each
/// update is a fresh decode composited under two `.ultraThinMaterial`
/// overlays. That is ordinary video-preview cost at this rate, but it is the
/// first thing to simplify if the send rate is raised much further.
struct ViewfinderCard: View {
    let frame: CapturedFrame?
    let isStreaming: Bool
    /// Why there is nothing to show yet, in plain language.
    let placeholderReason: String

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// The placeholder's glyph, scaled with the text beside it.
    @ScaledMetric(relativeTo: .title) private var placeholderSymbolSize: CGFloat = 40

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 18)
                .fill(Color(.secondarySystemGroupedBackground))

            if let frame {
                Image(uiImage: frame.image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    // Dimmed and desaturated once the stream stops. The frame
                    // is kept because it is the last thing the glasses really
                    // saw, but a full-brightness image in a large workspace
                    // layout reads as live - and the camera is off. The badge
                    // below says which, and this makes it legible at a glance.
                    .saturation(isStreaming ? 1 : 0.35)
                    .opacity(isStreaming ? 1 : 0.55)
                    .accessibilityLabel(
                        isStreaming
                            ? "Live frame from the glasses camera"
                            : "Last frame from the glasses camera. The camera is off."
                    )
            } else {
                placeholder
            }
        }
        // Compact until there is something to look at, so an idle dashboard
        // isn't dominated by an empty rectangle. The placeholder's height is
        // a minimum, so its sentence grows the card at the accessibility
        // sizes instead of being clipped by it.
        .frame(minHeight: frame == nil ? 150 : 260, maxHeight: frame == nil ? nil : 260)
        .clipShape(.rect(cornerRadius: 18))
        .overlay(alignment: .topLeading) {
            if isStreaming {
                liveBadge.padding(12)
            } else if let frame {
                // Replaces the LIVE badge rather than leaving the corner empty,
                // so a stopped preview always states what it is.
                stoppedBadge(sequence: frame.sequence).padding(12)
            }
        }
        .overlay(alignment: .bottomTrailing) {
            if let frame {
                Text("\(frame.width) × \(frame.height)")
                    .font(.caption2.weight(.medium).monospacedDigit())
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(.ultraThinMaterial, in: .capsule)
                    .padding(12)
            }
        }
    }

    private var placeholder: some View {
        VStack(spacing: 10) {
            Image(systemName: "eyeglasses")
                .font(.system(size: placeholderSymbolSize))
                .foregroundStyle(.tertiary)
                .accessibilityHidden(true)
            Text(placeholderReason)
                .font(.footnote)
                .foregroundStyle(.readableSecondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 32)
        }
        .padding(.vertical, 24)
        .accessibilityElement(children: .combine)
    }

    private func stoppedBadge(sequence: Int) -> some View {
        Text("Camera off · last frame #\(sequence)")
            .font(.caption2.weight(.medium))
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(.ultraThinMaterial, in: .capsule)
            .accessibilityHidden(true)
    }

    private var liveBadge: some View {
        HStack(spacing: 5) {
            Circle()
                .fill(.red)
                .frame(width: 7, height: 7)
                .opacity(reduceMotion ? 1 : 0.9)
            Text("LIVE")
                .font(.caption2.weight(.bold))
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(.ultraThinMaterial, in: .capsule)
        .accessibilityLabel("Live")
    }
}

#Preview("Empty") {
    ViewfinderCard(
        frame: nil,
        isStreaming: false,
        placeholderReason: "Start a session to see what the glasses see."
    )
    .padding()
}

#endif
