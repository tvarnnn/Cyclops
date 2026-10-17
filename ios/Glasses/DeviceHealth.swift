//
//  DeviceHealth.swift
//  Glasses
//

import Combine
import Foundation
import UIKit

/// iPhone-side health signals for the streaming session: thermal pressure,
/// Low Power Mode, and battery.
///
/// A five-minute physical run left the glasses noticeably warm, and a sender
/// rate that decays over minutes and a device that heats over minutes are the
/// same shape: without telemetry they are indistinguishable from each other —
/// and from a network that simply got worse. This says, after the fact,
/// whether the phone was throttling too.
///
/// Deliberately **only** the phone. The glasses have their own thermal
/// telemetry, but it arrives through Meta DAT, which by
/// ios/Glasses/Project_Overview_Steps/docs/02-DEVELOPMENT-RULES.md Rule 1 stays behind `GlassesConnection`; it is
/// published there as `glassesThermalLevel`. Nothing here talks to DAT.
///
/// Every value is read from the OS, never derived or estimated, and anything
/// the OS declines to report stays `nil` — Rule 3.
///
/// Note that constructing this switches on `UIDevice` battery monitoring
/// process-wide, and `ProjectManager` constructs it unconditionally while the
/// only consumer today is the DEBUG-only developer surface. Battery monitoring
/// is cheap, and the alternative — another `#if DEBUG` seam through
/// `ProjectManager` — buys less than it costs. Worth revisiting if this type
/// ever grows a more expensive observation.
@MainActor
final class DeviceHealth: ObservableObject {

    /// `.nominal`, `.fair`, `.serious`, `.critical`. At `.serious` the system
    /// is already reducing CPU/GPU availability, which would show up in this
    /// pipeline as the JPEG encode and the main-actor hop both growing.
    @Published private(set) var thermalState: ProcessInfo.ThermalState

    /// Low Power Mode throttles background work and networking, so a session
    /// that starts fine and degrades after the phone crosses 20% battery has a
    /// mundane explanation worth ruling out first.
    @Published private(set) var isLowPowerModeEnabled: Bool

    /// 0...1, or `nil` when the OS will not say. `UIDevice` reports `-1` when
    /// battery monitoring is off; that sentinel is translated to `nil` here
    /// rather than propagated, so no caller can mistake it for a reading.
    @Published private(set) var batteryLevel: Float?

    @Published private(set) var batteryState: UIDevice.BatteryState

    private var observationTasks: [Task<Void, Never>] = []

    init() {
        // Enabled before the first read, or `batteryLevel` returns -1 and
        // `batteryState` returns `.unknown`, and neither notification fires.
        UIDevice.current.isBatteryMonitoringEnabled = true

        // `thermalState` is read here before any observer is installed, which
        // Apple documents as a precondition for reliably receiving
        // `thermalStateDidChangeNotification` at all. The initialisation and
        // the requirement happen to coincide; the read is not incidental.
        thermalState = ProcessInfo.processInfo.thermalState
        isLowPowerModeEnabled = ProcessInfo.processInfo.isLowPowerModeEnabled
        batteryLevel = Self.currentBatteryLevel()
        batteryState = UIDevice.current.batteryState

        // One task per signal, each written out rather than routed through a
        // shared helper taking a closure: `Task { [weak self] in for await … }`
        // from a main-actor initialiser is the same shape `GlassesConnection`
        // already uses for its DAT streams, and it keeps every capture inside
        // the actor instead of handing a closure across the boundary.
        //
        // The `Notification` value is deliberately discarded and the underlying
        // property re-read: none of these notifications carry a payload, and
        // re-reading is both the documented usage and the one that does not
        // move a non-`Sendable` `Notification` anywhere.
        observationTasks.append(Task { [weak self] in
            for await _ in NotificationCenter.default.notifications(
                named: ProcessInfo.thermalStateDidChangeNotification
            ) {
                self?.thermalState = ProcessInfo.processInfo.thermalState
            }
        })
        // Note the spelling: this name lives on `NSNotification.Name`, not on
        // `ProcessInfo` alongside the thermal one. There is no
        // `ProcessInfo.lowPowerModeDidChangeNotification`.
        observationTasks.append(Task { [weak self] in
            for await _ in NotificationCenter.default.notifications(
                named: .NSProcessInfoPowerStateDidChange
            ) {
                self?.isLowPowerModeEnabled = ProcessInfo.processInfo.isLowPowerModeEnabled
            }
        })
        observationTasks.append(Task { [weak self] in
            for await _ in NotificationCenter.default.notifications(
                named: UIDevice.batteryLevelDidChangeNotification
            ) {
                self?.batteryLevel = Self.currentBatteryLevel()
            }
        })
        observationTasks.append(Task { [weak self] in
            for await _ in NotificationCenter.default.notifications(
                named: UIDevice.batteryStateDidChangeNotification
            ) {
                self?.batteryState = UIDevice.current.batteryState
            }
        })
    }

    /// `isolated` so it may touch main-actor state, matching
    /// `GlassesConnection.deinit`. A plain `deinit` runs outside the actor and
    /// could not read `observationTasks` at all.
    isolated deinit {
        for task in observationTasks { task.cancel() }
    }

    /// Re-reads every value now.
    ///
    /// `batteryLevelDidChangeNotification` is documented to fire at most once a
    /// minute and in ~5% steps, so a session shorter than that can otherwise
    /// show a battery figure from before it started. Called when a session
    /// begins so the numbers on screen belong to the run being measured.
    func refresh() {
        thermalState = ProcessInfo.processInfo.thermalState
        isLowPowerModeEnabled = ProcessInfo.processInfo.isLowPowerModeEnabled
        batteryLevel = Self.currentBatteryLevel()
        batteryState = UIDevice.current.batteryState
    }

    private static func currentBatteryLevel() -> Float? {
        let level = UIDevice.current.batteryLevel
        return level < 0 ? nil : level
    }
}

extension ProcessInfo.ThermalState {
    /// Human-readable, and stable regardless of how the enum is spelled — the
    /// developer surface shows this rather than a raw case dump.
    var displayName: String {
        switch self {
        case .nominal: return "Nominal"
        case .fair: return "Fair"
        case .serious: return "Serious"
        case .critical: return "Critical"
        @unknown default: return "Unknown"
        }
    }

    /// Whether the system has begun shedding performance. At `.serious` and
    /// above, a send-rate shortfall may be the phone rather than the network,
    /// which is the distinction this whole type exists to make.
    var isThrottling: Bool {
        switch self {
        case .nominal, .fair: return false
        case .serious, .critical: return true
        @unknown default: return false
        }
    }
}

extension UIDevice.BatteryState {
    var displayName: String {
        switch self {
        case .unknown: return "Unknown"
        case .unplugged: return "On battery"
        case .charging: return "Charging"
        case .full: return "Full"
        @unknown default: return "Unknown"
        }
    }
}
