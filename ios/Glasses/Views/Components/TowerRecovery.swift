//
//  TowerRecovery.swift
//  Glasses
//

import SwiftUI

/// The words of the Tower's recovery controls (U0.8 F03).
enum TowerRecoveryText {
    static let connect = "Connect to Tower"
    static let connecting = "Connecting…"
    static let settings = "Tower settings"
}

/// What a "Not connected" panel can offer: a person's Connect, and the two
/// sheets where the Tower is recovered. Handed down from the root through
/// the environment, so a panel deep in a workspace needs no new parameter.
struct TowerRecoveryActions {
    let tower: TowerClient
    let openConnections: () -> Void
    let openSettings: () -> Void
}

extension EnvironmentValues {
    /// `nil` where no root supplied it -- previews and tests -- and a panel
    /// then draws no controls rather than inert ones.
    @Entry var towerRecovery: TowerRecoveryActions? = nil
}

/// Connect and Tower settings, side by side (stacked at the accessibility
/// sizes, as `FailureBanner` does).
///
/// A leaf that observes the Tower, the way `TowerReachabilityReader` does,
/// so the workspace above it does not re-render at the Tower's reply rate.
/// `connect()` is the same call Connections makes: it refills the reconnect
/// budget, which is right for a person's tap.
struct TowerRecoveryButtons: View {
    let actions: TowerRecoveryActions
    /// `<prefix>-connect` and `<prefix>-settings`, so two sets on one screen
    /// (a panel's and the capture control's) can be told apart.
    let identifierPrefix: String
    @ObservedObject private var tower: TowerClient
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    init(actions: TowerRecoveryActions, identifierPrefix: String = "tower-recovery") {
        self.actions = actions
        self.identifierPrefix = identifierPrefix
        _tower = ObservedObject(wrappedValue: actions.tower)
    }

    var body: some View {
        let layout = dynamicTypeSize.isAccessibilitySize
            ? AnyLayout(VStackLayout(alignment: .leading, spacing: 8))
            : AnyLayout(HStackLayout(spacing: 10))
        layout {
            Button(tower.status == .connecting ? TowerRecoveryText.connecting : TowerRecoveryText.connect) {
                tower.connect()
            }
            .buttonStyle(.borderedProminent)
            .disabled(tower.status == .connecting || tower.status == .online)
            .accessibilityIdentifier("\(identifierPrefix)-connect")

            Button(TowerRecoveryText.settings) { actions.openSettings() }
                .readableBorderedButton()
                .accessibilityIdentifier("\(identifierPrefix)-settings")
        }
        .font(.subheadline.weight(.medium))
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
