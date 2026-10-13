//
//  ViewfinderCard.swift
//  Glasses
//

// `CapturedFrame` and the whole camera path are DEBUG-only in the model
// (Glasses/GlassesConnection.swift). This entire file is gated to match, so no
// control ever crosses the conditional boundary.
#if DEBUG

// For `PermissionStatus`, read by `ViewfinderText.placeholder`.
import MWDATCore
import SwiftUI

/// What the viewfinder says when there is no frame to show, and what its
/// corner badge says (U0.8 F05). One place, so World Builder and Home cannot
/// drift apart, and so no state says "Start…" while the camera is on.
enum ViewfinderText {
    static let waitingForGlasses = "Waiting for the glasses to become active."
    static let waitingForFirstFrame = "Camera on. Waiting for the first frame from the glasses."
    static let starting = "Starting the glasses camera…"
    static let paused = "The glasses paused the capture. It resumes on its own, and this app cannot override that."
    static let waitingBadge = "Camera on · no frame yet"
    static let pausedBadge = "Paused by the glasses"

    /// `noun` is "capture session" (World Builder) or "session" (Home).
    static func placeholder(isStreaming: Bool, isEngaged: Bool, isPausedByGlasses: Bool,
                            hasActiveDevice: Bool, permission: PermissionStatus?, noun: String) -> String {
        if isStreaming { return waitingForFirstFrame }
        if isEngaged { return starting }
        if isPausedByGlasses { return paused }
        if !hasActiveDevice { return waitingForGlasses }
        if permission != .granted { return "Camera access is needed before a session can stream." }
        return "Start a \(noun) to see what the glasses see."
    }

    /// Under Stop while the glasses hold the capture paused (U0.8 F06).
    /// `stopTitle` is the control's own word: "Stop capture" or "Stop session".
    static func pausedControlLine(stopTitle: String) -> String {
        "The glasses paused this capture. It resumes on its own; \(stopTitle) ends it."
    }
}

/// Which capture control a workspace draws (U0.8 F06). A capture the glasses
/// paused is still claimed -- a Start then is refused and does nothing
/// observable -- so it offers Stop, never Start.
enum CaptureControlMode: Equatable {
    case stop
    case stopWhilePaused
    case start

    static func mode(isEngaged: Bool, claim: CaptureClaim) -> CaptureControlMode {
        if isEngaged { return .stop }
        if claim == .devicePaused { return .stopWhilePaused }
        return .start
    }
}

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
    /// The glasses paused the capture themselves (`CaptureClaim.devicePaused`).
    var isPausedByGlasses: Bool = false

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
                            : isPausedByGlasses
                                ? "Last frame from the glasses camera. The glasses paused the capture."
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
            // LIVE only over a frame: a stream with nothing to show yet says
            // so rather than badging an empty card as live.
            if isStreaming && frame != nil {
                liveBadge.padding(12)
            } else if isStreaming {
                textBadge(ViewfinderText.waitingBadge).padding(12)
            } else if isPausedByGlasses {
                textBadge(ViewfinderText.pausedBadge + (frame.map { " · last frame #\($0.sequence)" } ?? ""))
                    .padding(12)
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

    /// A state in words, read by VoiceOver: unlike the stopped badge, the
    /// card's other words do not already say it.
    private func textBadge(_ text: String) -> some View {
        Text(text)
            .font(.caption2.weight(.medium))
            .foregroundStyle(.readableSecondary)
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(.ultraThinMaterial, in: .capsule)
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
