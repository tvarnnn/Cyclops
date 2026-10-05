//
//  CaptureHealthView.swift
//  Glasses
//
//  U2-D0 "Capture health": the operator's panel in the capture screen, in
//  large type. Five rows, each one VoiceOver element; the figures come from
//  `CaptureHealthModel` and are re-read once a second, so a stall's seconds
//  count up and a report that stopped arriving turns into "—" rather than
//  staying on screen as if it were live. It claims no coverage and no
//  position.
//

import SwiftUI

struct CaptureHealthView: View {
    @ObservedObject var model: CaptureHealthModel
    /// The Tower pill's link (`TowerClient.status == .online`).
    let isLinked: Bool
    /// A capture is running on this phone: only then can it be "stalled".
    let isCapturing: Bool

    @State private var announcer = CaptureHealthAnnouncer()

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { _ in
            let readout = model.history.readout(now: .now, isLinked: isLinked, isCapturing: isCapturing)
            rows(readout)
                .onChange(of: readout) { old, new in announce(from: old, to: new) }
        }
    }

    private func rows(_ readout: CaptureHealthReadout) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            SectionLabel("Capture health")
            row(readout.link, id: "capture-health-link")
            row(readout.pace, id: "capture-health-pace")
            if let stalled = readout.stalled {
                row(stalled, id: "capture-health-stalled")
            }
            row(readout.breaks, spoken: readout.breaksSpoken, id: "capture-health-breaks")
            lookBack(readout)
            if let map = readout.mapLag {
                row(map, id: "capture-health-map")
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 12).fill(Color(.secondarySystemBackground)))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("capture-health")
    }

    private func row(_ text: String, spoken: String? = nil, id: String) -> some View {
        Text(text)
            .font(.title3.monospacedDigit())
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
            .dynamicTypeSize(...DynamicTypeSize.accessibility1)
            .accessibilityShowsLargeContentViewer { Text(text) }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(spoken ?? text)
            .accessibilityIdentifier(id)
    }

    /// The relocalizer's line (the canvas's words), then its counts: one row,
    /// one element.
    private func lookBack(_ readout: CaptureHealthReadout) -> some View {
        let text = [readout.lookBackLine, readout.lookBackCounts].compactMap { $0 }.joined(separator: "\n")
        return row(text, spoken: text.replacingOccurrences(of: "\n", with: ". "), id: "capture-health-lookback")
    }

    private func announce(from old: CaptureHealthReadout, to new: CaptureHealthReadout) {
        guard let sentence = announcer.announcement(from: old, to: new, at: .now) else { return }
        AccessibilityNotification.Announcement(sentence).post()
    }
}
