//
//  IMURecordingBadge.swift
//  Glasses
//

// DEBUG-only, like the recorder it reports on.
#if DEBUG

import SwiftUI
import UIKit

/// The persistent sign that this phone is writing an IMU log.
///
/// Motion is raw sensor data (docs/06-PRIVACY-DATA.md), so while the recorder
/// holds a file this badge sits in the navigation bar above every workspace,
/// where no cartridge can hide it and no scroll moves it. It observes only the
/// recorder's readout, not `GlassesConnection`, so it redraws with the
/// recorder rather than with every camera frame. Nothing shows when no
/// recorder is open.
struct IMURecordingBadge: View {
    @ObservedObject var readout: IMURecorderReadout

    var body: some View {
        switch readout.indicator {
        case .off:
            EmptyView()
        case .writing:
            badge(symbol: "record.circle.fill", title: "IMU", tint: .red, value: "recording")
        case .stopped(let reason):
            // The reason is on the badge itself, not only in its accessibility
            // value: "IMU protection" (refused), "IMU error", "IMU duration cap".
            badge(symbol: "exclamationmark.triangle.fill", title: "IMU \(reason)", tint: .orange, value: "stopped, \(reason)")
        }
    }

    /// The badge's words: the label colour on the tinted capsule, like the
    /// drawer's "Ready to test" badge. The tint's own red or orange on its
    /// 15 % capsule was about 2:1 to 3:1 (U0.5 review F13), under the 4.5:1
    /// text needs; the capsule and the symbol still carry the colour.
    static let textColor = UIColor.label
    /// The capsule's share of the tint.
    static let capsuleOpacity: CGFloat = 0.15

    private func badge(symbol: String, title: String, tint: Color, value: String) -> some View {
        Label {
            Text(title)
                .foregroundStyle(Color(uiColor: Self.textColor))
        } icon: {
            Image(systemName: symbol)
                .foregroundStyle(tint)
        }
            .labelStyle(.titleAndIcon)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(tint.opacity(Self.capsuleOpacity), in: .capsule)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("IMU log")
            .accessibilityValue(value)
    }
}

#endif
