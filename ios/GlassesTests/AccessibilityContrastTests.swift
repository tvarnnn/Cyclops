//
//  AccessibilityContrastTests.swift
//  GlassesTests
//
//  U0.5: the colours the accessibility pass chose, measured. WCAG 2 contrast
//  ratios of the readable secondary text on every surface it sits on, of the
//  viewer's dark loading panel, and of the look-back banner, which the
//  Simulator cannot reach (it needs a live walk) and so the UI audit cannot
//  see. Plus the two pieces of text handling: the override note's break
//  points and the loading panel's progress line.
//

import SwiftUI
import UIKit
import XCTest

@testable import Glasses

final class AccessibilityContrastTests: XCTestCase {

    // MARK: Traits

    private let light = UITraitCollection(userInterfaceStyle: .light)
    private let dark = UITraitCollection(userInterfaceStyle: .dark)
    /// Sheets (Connections, Settings, the drawer, Saved worlds) draw their
    /// cards at the elevated level, which in dark mode is a lighter grey.
    private var darkElevated: UITraitCollection {
        dark.modifyingTraits { $0.userInterfaceLevel = .elevated }
    }

    // MARK: Readable secondary text

    func testReadableSecondaryIsAtLeast4Point5To1OnEverySurfaceInLight() {
        let text = UIColor.readableSecondary
        for background in [UIColor.secondarySystemGroupedBackground, .systemGroupedBackground,
                           .systemBackground] {
            assertContrast(text, on: background, traits: light, atLeast: 4.5)
        }
        // A tinted capsule over a card: the drawer's "Not built" badge.
        assertContrast(text, on: .tertiarySystemFill, over: .secondarySystemGroupedBackground,
                       traits: light, atLeast: 4.5)
    }

    func testReadableSecondaryIsAtLeast4Point5To1OnEverySurfaceInDark() {
        let text = UIColor.readableSecondary
        for traits in [dark, darkElevated] {
            for background in [UIColor.secondarySystemGroupedBackground, .systemGroupedBackground,
                               .systemBackground] {
                assertContrast(text, on: background, traits: traits, atLeast: 4.5)
            }
            assertContrast(text, on: .tertiarySystemFill, over: .secondarySystemGroupedBackground,
                           traits: traits, atLeast: 4.5)
        }
    }

    /// What it replaced, so the reason for it stays on record: the system's
    /// tertiary label fails on a card in both appearances, and the secondary
    /// label fails on white.
    func testTheSystemGreysItReplacesFallShort() {
        XCTAssertLessThan(ratio(.tertiaryLabel, on: .secondarySystemGroupedBackground, traits: light), 4.5)
        XCTAssertLessThan(ratio(.tertiaryLabel, on: .secondarySystemGroupedBackground, traits: dark), 4.5)
        XCTAssertLessThan(ratio(.secondaryLabel, on: .secondarySystemGroupedBackground, traits: light), 4.5)
    }

    func testIncreaseContrastUsesTheLabelColour() {
        let high = light.modifyingTraits { $0.accessibilityContrast = .high }
        let resolved = UIColor.readableSecondary.resolvedColor(with: high)
        XCTAssertEqual(resolved, UIColor.label.resolvedColor(with: high))
    }

    // MARK: Tinted labels

    func testTheReadableTintIsAtLeast4Point5To1OnItsCapsuleAndOnCards() {
        for traits in [light, dark, darkElevated] {
            let tint = UIColor.readableTint.resolvedColor(with: traits)
            // A bordered button's capsule: the tint at about 15 % over the page.
            let capsule = tint.withAlphaComponent(0.15)
            assertContrast(tint, on: capsule, over: .systemGroupedBackground, traits: traits, atLeast: 4.5)
            assertContrast(tint, on: capsule, over: .secondarySystemGroupedBackground, traits: traits, atLeast: 4.5)
            assertContrast(tint, on: .secondarySystemGroupedBackground, traits: traits, atLeast: 4.5)
        }
    }

    func testTheReadableDestructiveIsAtLeast4Point5To1OnCards() {
        for traits in [light, dark, darkElevated] {
            assertContrast(.readableDestructive, on: .secondarySystemGroupedBackground, traits: traits, atLeast: 4.5)
        }
    }

    // MARK: The viewer's loading panel

    func testTheLoadingPanelIsThePagesOwnBackground() {
        var red: CGFloat = 0, green: CGFloat = 0, blue: CGFloat = 0, alpha: CGFloat = 0
        WorldRenderLoadingPanel.pageBackground.getRed(&red, green: &green, blue: &blue, alpha: &alpha)
        XCTAssertEqual([red, green, blue, alpha].map { Int(($0 * 255).rounded()) }, [0x0B, 0x0D, 0x10, 255],
                       "`--bg` in the Tower's viewer pages")
    }

    func testTheLoadingPanelTextIsAtLeast4Point5To1() {
        // The panel forces the dark scheme, so its text resolves dark.
        let background = WorldRenderLoadingPanel.pageBackground
        assertContrast(.label, on: background, traits: dark, atLeast: 4.5)
        assertContrast(UIColor.readableSecondary, on: background, traits: dark, atLeast: 4.5)
    }

    func testTheLoadingPanelSaysWhichStepAndHowLong() {
        XCTAssertEqual(WorldRenderLoadingPanel.progress(step: 1, seconds: 0), "Step 1 of 2")
        XCTAssertEqual(WorldRenderLoadingPanel.progress(step: 1, seconds: 1), "Step 1 of 2")
        XCTAssertEqual(WorldRenderLoadingPanel.progress(step: 2, seconds: 7), "Step 2 of 2 · 7 s")
    }

    // MARK: The look-back banner

    /// Title 3 semibold is large text, which needs 3:1; held to 4.5:1 anyway,
    /// because it is read at a glance mid-walk. White managed about 2.3:1.
    func testTheLookBackBannerTextIsAtLeast4Point5To1() {
        let text = WorldLookBackBannerView.textColor
        for traits in [light, dark] {
            assertContrast(text, on: WorldLookBackBannerView.lookBackFill, traits: traits, atLeast: 4.5)
            assertContrast(text, on: WorldLookBackBannerView.backOnTrackFill, traits: traits, atLeast: 4.5)
        }
        XCTAssertLessThan(ratio(.white, on: WorldLookBackBannerView.lookBackFill, traits: light), 3,
                          "the white it replaced")
    }

    // MARK: The IMU badge (DEBUG)

    #if DEBUG
    /// The IMU log's badge in the navigation bar, which the Simulator cannot
    /// reach (it needs glasses) and so the UI audit cannot see: its words on
    /// its red or orange capsule, over the bar in either appearance (U0.5
    /// review F13).
    func testTheIMUBadgeTextIsAtLeast4Point5To1OnItsCapsule() {
        for traits in [light, dark] {
            for tint in [UIColor.systemRed, .systemOrange] {
                let capsule = tint.resolvedColor(with: traits).withAlphaComponent(IMURecordingBadge.capsuleOpacity)
                assertContrast(IMURecordingBadge.textColor, on: capsule, over: .systemBackground,
                               traits: traits, atLeast: 4.5)
            }
        }
        let orange = UIColor.systemOrange.withAlphaComponent(IMURecordingBadge.capsuleOpacity)
        XCTAssertLessThan(ratio(.systemOrange, on: orange, over: .systemBackground, traits: light), 3,
                          "the orange text it replaced")
    }
    #endif

    // MARK: The override note's variable name

    func testTheVariableNameBreaksOnlyAtItsUnderscores() {
        let shown = TowerSettingsText.breakable("from GLASSES_TOWER_AUTHORITY, a developer setting")
        XCTAssertEqual(shown, "from GLASSES_\u{200B}TOWER_\u{200B}AUTHORITY, a developer setting")
        XCTAssertEqual(shown.replacingOccurrences(of: "\u{200B}", with: ""),
                       "from GLASSES_TOWER_AUTHORITY, a developer setting", "nothing else changes")
        XCTAssertEqual(TowerSettingsText.breakable("no underscore"), "no underscore")
        // An address and a contract break at their own punctuation; a full
        // stop that ends a sentence gets nothing.
        XCTAssertEqual(TowerSettingsText.breakable("127.0.0.1:63219"),
                       "127.\u{200B}0.\u{200B}0.\u{200B}1:\u{200B}63219")
        XCTAssertEqual(TowerSettingsText.breakable("cartridge_results.envelope/2026-08-23"),
                       "cartridge_\u{200B}results.\u{200B}envelope/\u{200B}2026-08-23")
        XCTAssertEqual(TowerSettingsText.breakable("Saved. Glasses"), "Saved. Glasses")
    }

    // MARK: Measuring

    private func assertContrast(_ text: UIColor, on background: UIColor, over base: UIColor? = nil,
                                traits: UITraitCollection, atLeast minimum: Double,
                                file: StaticString = #filePath, line: UInt = #line) {
        let measured = ratio(text, on: background, over: base, traits: traits)
        XCTAssertGreaterThanOrEqual(
            measured, minimum,
            "\(text) on \(background) (\(traits.userInterfaceStyle == .dark ? "dark" : "light")): \(measured)",
            file: file, line: line
        )
    }

    /// WCAG 2 contrast of `text` over `background`, itself over `base` (or
    /// opaque), all resolved for `traits`.
    private func ratio(_ text: UIColor, on background: UIColor, over base: UIColor? = nil,
                       traits: UITraitCollection) -> Double {
        let baseRGB = base.map { rgba($0, traits).rgb } ?? [1, 1, 1]
        let back = composite(rgba(background, traits), over: baseRGB)
        let fore = composite(rgba(text, traits), over: back)
        let (l1, l2) = (luminance(fore), luminance(back))
        return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)
    }

    private func rgba(_ color: UIColor, _ traits: UITraitCollection) -> (rgb: [Double], alpha: Double) {
        var red: CGFloat = 0, green: CGFloat = 0, blue: CGFloat = 0, alpha: CGFloat = 0
        color.resolvedColor(with: traits).getRed(&red, green: &green, blue: &blue, alpha: &alpha)
        return ([red, green, blue].map { min(max(Double($0), 0), 1) }, Double(alpha))
    }

    private func composite(_ top: (rgb: [Double], alpha: Double), over bottom: [Double]) -> [Double] {
        zip(top.rgb, bottom).map { $0 * top.alpha + $1 * (1 - top.alpha) }
    }

    private func luminance(_ rgb: [Double]) -> Double {
        let linear = rgb.map { $0 <= 0.04045 ? $0 / 12.92 : pow(($0 + 0.055) / 1.055, 2.4) }
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    }
}
