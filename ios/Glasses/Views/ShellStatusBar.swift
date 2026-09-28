//
//  ShellStatusBar.swift
//  Glasses
//

import MWDATCore
import SwiftUI

#if DEBUG
import MWDATCamera
#endif

/// The persistent infrastructure readout: glasses, camera, Tower.
///
/// Lives in the shell, above the workspace and outside the workspace switch, so
/// no cartridge can hide it and no workspace change can rebuild it. That is
/// deliberate and privacy-relevant: because leaving a workspace does *not* stop
/// a running camera, the camera's state has to be visible from everywhere,
/// including from Home after the user has navigated away from the workspace
/// that started it.
///
/// Tapping opens `ConnectionSheet`, which is where every manual recovery
/// control now lives. Keeping that route open matters — the Tower's automatic
/// reconnect schedule is bounded and deliberately gives up, so there has to be
/// a way for a person to say "try again".
struct ShellStatusBar: View {
    @ObservedObject var glasses: GlassesConnection
    @ObservedObject var tower: TowerClient
    let onTap: () -> Void

    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    /// Three across, or -- at the accessibility sizes, where a third of the
    /// screen is too narrow for "Connected" -- one above the other, each
    /// value with the full width.
    private var isStacked: Bool { dynamicTypeSize.isAccessibilitySize }

    var body: some View {
        let layout = isStacked
            ? AnyLayout(VStackLayout(spacing: 8))
            : AnyLayout(HStackLayout(spacing: 10))
        let pill: StatusPill.Layout = isStacked ? .row : .column
        Button(action: onTap) {
            layout {
                StatusPill(
                    title: "Glasses",
                    value: StateDisplay.registration(glasses.registrationState),
                    level: glassesLevel,
                    layout: pill
                )
                #if DEBUG
                StatusPill(
                    title: "Camera",
                    value: cameraValue,
                    level: cameraLevel,
                    layout: pill
                )
                #endif
                StatusPill(
                    title: "Tower",
                    value: StateDisplay.tower(tower.status),
                    level: towerLevel,
                    layout: pill
                )
            }
        }
        .buttonStyle(.plain)
        .accessibilityHint("Opens connection settings")
    }

    private var glassesLevel: StatusLevel {
        switch glasses.registrationState {
        case .registered: return .ok
        case .registering: return .working
        case .unavailable: return .problem
        default: return .idle
        }
    }

    private var towerLevel: StatusLevel {
        switch tower.status {
        case .online: return .ok
        case .connecting: return .working
        case .failed: return .problem
        case .offline: return .idle
        }
    }

    #if DEBUG
    /// "Camera on"/"Camera off" rather than the DAT case name.
    ///
    /// The question a person is asking of this pill is whether the glasses are
    /// recording, and `Stopped` is a poor answer to it. The transitional states
    /// keep their own wording because "off" would be wrong while a stream is
    /// coming up.
    private var cameraValue: String {
        switch glasses.cameraStreamState {
        // The CV Lab's frame hold is a fact about the socket, not the
        // camera, but it is set on one screen and the camera is shown on
        // every screen; a pill reading "On" over a Home whose "Sent to
        // Tower" figure is decaying would be the shell not knowing.
        case .streaming: return tower.isFrameSendingPaused ? "On · held" : "On"
        case .stopped: return "Off"
        default: return StateDisplay.cameraStream(glasses.cameraStreamState)
        }
    }

    private var cameraLevel: StatusLevel {
        switch glasses.cameraStreamState {
        case .streaming: return .ok
        case .starting, .stopping, .waitingForDevice: return .working
        case .paused, .stopped: return .idle
        @unknown default: return .idle
        }
    }
    #endif
}
